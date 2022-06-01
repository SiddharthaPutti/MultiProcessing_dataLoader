import hydra
from omegaconf import DictConfig, OmegaConf
import time
import concurrent.futures
import load_dataset


@hydra.main(config_path="./configs", config_name="data.yaml")
def main(config: DictConfig) -> None:
    # from model import model
    dataset = load_dataset.make_batches(config)
    data = []
    data_labels = []
    # print(len(data_batches))
    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_batches = [executor.submit(load_dataset.loader, batches) for batches in dataset]

        for result in concurrent.futures.as_completed(result_batches):
            data.extend(result.result()[0])
            data_labels.extend(result.result()[1])
    print(len(data))

    end = time.perf_counter()
    print('time taken to load dataset :', end - start)
    #model.main(data, data_labels)





if __name__ == '__main__':
    main()