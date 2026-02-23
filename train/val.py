from torch.utils.data import DataLoader
import torch.nn.functional as F


def val_loop(data_dict, idxs, model_classifier, dataset_classifier, device, dataset_extractor=None, model_extractor=None, batch_size=12):

    if model_extractor!=None:
        vd =dataset_extractor(data_dict, idxs)
        valoader_extractor = DataLoader(vd, batch_size=batch_size, shuffle=True, pin_memory=False)
        del vd
        model_extractor.to(device)
        model_extractor.eval()


        last_dims, affectation_rec, hospitals_rec, patients_rec, slides_rec, coords_rec, n_padding_rec = [], [], [], [], [], [], []
        for b in valoader_extractor:
            x, y, extra_info = b

            # Extra info deglossed:
            hosps_batch = extra_info['hospital']
            pats_batch = extra_info['patient']
            slds_batch = extra_info['slides']
            coords_batch = extra_info['coords']
            affectation_batch = extra_info['coords']


            output, last_dim = model_extractor(x.to(device))

            # losses.append(loss.item())
            last_dim_cpu =last_dim.cpu().detach()
            last_dims.extend(last_dim_cpu)
            affectation_rec.extend(affectation_batch)
            hospitals_rec.extend(hosps_batch)
            patients_rec.extend(pats_batch)
            slides_rec.extend(slds_batch)
            coords_rec.extend(coords_batch)


        rebuild_params ={
            'features': last_dims,
            'affectation': affectation_rec,
            'hospitals': hospitals_rec,
            'patients': patients_rec,
            'slides': slides_rec,
            'coords': coords_rec,
        }

        data_dict =patient_dict_builder(**rebuild_params)

        model_classifier.to(device)
        model_classifier.eval()

        vd = dataset_classifier(data_dict, idxs)

        valoader_classifier = DataLoader(vd, batch_size=batch_size, shuffle=True, pin_memory=False)
        del vd

        attention_output = {}

        y_true, y_pred, y_scores = [], [], []

        for b in valoader_classifier:
            x, y, extra_info, histodata = b

            # Extra info deglossed:
            hosps_batch = extra_info['hospital']
            pats_batch = extra_info['patient']
            slds_batch = extra_info['slides']
            coords_batch = extra_info['coords']
            affectation_batch = extra_info['coords']
            n_padding_batch = extra_info['n_padding']

            output, attention_scores = model_classifier(x.to(device), n_padding=n_padding_batch, histodata=None)

            probs = F.softmax(output, dim=1).cpu()

            y_true.extend(y.cpu().tolist())

            y_pred.extend(probs.argmax(dim=1).cpu().tolist())
            y_scores.extend(probs[:, 1].cpu().tolist())

            # storing attention
            for i, hospi in enumerate(hosps_batch):

                hosp = hosps_batch[i]
                pat = pats_batch[i]
                true = y[i].float().item()
                label_p = true
                st_probs = probs.cpu().tolist()[i]
                pred = probs.argmax(dim=1).cpu().tolist()[i]
                pred = float(pred)
                n_pad = n_padding_batch[i]
                n_pad = mx_patches - n_pad

                slide_ds = slds_batch[i]
                attention_matrixs = attention_scores[i].cpu()
                coord_ds = coords_batch[i].detach().cpu()
                x_ds = x[i].detach().cpu()

                attention_matrixs = attention_matrixs[:n_pad]

                coord_ds = coord_ds[:n_pad]

                slide_ds = slide_ds[:n_pad]
                x_ds = x_ds[:n_pad]

                for s, slide_s in enumerate(slide_ds):
                    slide = slide_ds[s]
                    slide = slide.item()
                    slide = inv_slide_index_dict[slide]
                    coord_d = coord_ds[s].numpy()
                    features = x_ds[s].numpy()

                    # attention_matrix=attention_matrixs[0][s].detach().item() #Zero [0] so it doesnt fuck up everything

                    attention_matrix_detached = attention_matrixs.detach()
                    attention_matrix_heads = {}
                    for a, att_head in enumerate(attention_matrix_detached):
                        attention_matrix_heads[a] = attention_matrix_detached[a][s]

                    if hosp not in attention_output:
                        attention_output[hosp] = {}

                    if pat not in attention_output[hosp]:
                        attention_output[hosp][pat] = {}

                    if slide not in attention_output[hosp][pat]:
                        attention_output[hosp][pat][slide] = []

                    attention_output[hosp][pat]['scores'] = st_probs
                    attention_output[hosp][pat]['label'] = true
                    attention_output[hosp][pat]['predicted'] = pred
                    attention_output[hosp][pat][slide].append((features, coord_d, attention_matrix_heads))

    return y_true, y_pred, y_scores, attention_output