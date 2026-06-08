import sys
from pathlib import Path
from torch_geometric.loader import DataLoader

# Add src to path for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent / "src") not in sys.path:
    sys.path.append(str(current_dir.parent / "src"))

from dataset.graph_loaders import GraphDataset

def test_loading():
    # 1. Configuración de rutas
    # Ajusta esta ruta a donde realmente tengas los grafos (ej: KNN, Radius o Fully Connected)
    graphs_dir = current_dir.parent / "data" / "graphs" / "knn" / "cosine" / "k_8"
    features_dir = current_dir.parent / "data" / "graphs" / "features"

    print(f"--- Iniciando prueba de carga de grafos desde: {graphs_dir} ---")

    if not graphs_dir.exists():
        print(f"ERROR: La carpeta de grafos {graphs_dir} no existe.")
        return

    # 2. Obtener lista de pacientes (necesitamos los IDs que tienen archivos .pt)
    # Podemos leerlos directamente de la carpeta de grafos
    patient_files = list(graphs_dir.glob("*.pt"))
    if not patient_files:
        print("ERROR: No se encontraron archivos .pt en la carpeta.")
        return
    
    patient_ids = [f.stem for f in patient_files]
    print(f"Se encontraron {len(patient_ids)} pacientes con grafos.")

    # 3. Instanciar el Dataset
    dataset = GraphDataset(graphs_dir, patient_ids, features_dir=features_dir)

    # 4. Probar el DataLoader de PyTorch Geometric
    # Este paso es CRÍTICO: verifica que el batching especial de grafos funciona
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    try:
        # Obtenemos el primer batch
        batch = next(iter(loader))
        print("\n--- Batch Cargado Correctamente ---")
        print(f"Número de grafos en el batch: {batch.num_graphs}")
        print(f"Atributos del batch: {batch.keys()}")
        print(f"Forma de las características (nodos_totales, features): {batch.x.shape}")
        print(f"Forma del índice de aristas (2, aristas_totales): {batch.edge_index.shape}")
        print(f"Etiquetas del batch: {batch.y}")
        print(f"Vector de batch (asigna cada nodo a un grafo): {batch.batch.shape}")
        
        # Verificar un grafo individual
        sample_graph = dataset[0]
        print("\n--- Detalle de un Grafo Individual ---")
        print(f"Nodos: {sample_graph.num_nodes}")
        print(f"Aristas: {sample_graph.num_edges}")
        
        if hasattr(sample_graph, 'histodata'):
             print(f"Datos histológicos presentes: {sample_graph.histodata.shape}")

        print("\n¡Prueba superada con éxito! La capa de datos está lista.")

    except Exception as e:
        print(f"\nERROR durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
