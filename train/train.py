from torch.utils.data import DataLoader




def train_loop(data_dict, idxs, model_classifier, dataset_classifier, loss_function_classifier, device, optimizer_classifier, model_extractor=None, dataset_extractor=None, loss_function_extractor=None, optimizer_extractor=None, epochs=30, minibatch_size=2, batch_size=12):

    if model_extractor!=None:
        td =dataset_extractor(data_dict, idxs)
        ppp =len(td)
        trainloader_extractor = DataLoader(td, batch_size=batch_size, shuffle=True, pin_memory=False)
        del td
        model_extractor.to(device)
        model_extractor.train()
        model_classifier.eval()
        loss_function_extractor.to(device)

        for epoch in range(epochs):

            batch_counter = 0

            last_dims, affectation_rec, hospitals_rec, patients_rec, slides_rec, coords_rec, = [], [], [], [], [], []

            for b in trainloader_extractor:
                x, y, extra_info = b

                # Extra info deglossed:
                hosps_batch = extra_info['hospital']
                pats_batch = extra_info['patient']
                slds_batch = extra_info['slides']
                coords_batch = extra_info['coords']
                affectation_batch = extra_info['coords']


                output, last_dim = model_extractor(x.to(device))

                loss_ex = loss_function_extractor(output.float(), y.long().to(device))
                # loss_ex = loss_ex / minibatch_size  # Scale loss
                loss_ex.backward()


                batch_counter +=1
                if batch_counter >= minibatch_size:
                    optimizer_extractor.step()
                    optimizer_extractor.zero_grad()
                    batch_counter = 0

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

        model_extractor.eval()

    model_classifier.train()
    model_classifier.to(device)
    loss_function_classifier.to(device)
    td =dataset_classifier(data_dict, idxs)
    trainloader_classifier = DataLoader(td, batch_size=batch_size, shuffle=True, pin_memory=False)
    del td


    for epoch in range(epochs):

        batch_counter = 0
        for b in trainloader_classifier:
            x, y, extra_info, histodata = b

            n_padding_batch =extra_info['n_padding']
            output, _ = model_classifier(x.to(device), n_padding=n_padding_batch, histodata=None)

            loss_clf = loss_function_classifier(output, y.long().to(device))
            loss_clf = loss_clf / minibatch_size  # Scale loss
            loss_clf.backward()

            batch_counter +=1
            if batch_counter >= minibatch_size:
                optimizer_classifier.step()
                optimizer_classifier.zero_grad()
                batch_counter = 0

    if model_extractor!=None:
        trained_models =[model_extractor, model_classifier]

    else:
        trained_models =[model_classifier]
    return trained_models