import csv
import argparse
import torch
from torch.utils.data import DataLoader
from logger import CompleteLogger
from dataload_rarp import CustomVideoDataset
from eval_utils import compute_binary_metrics, format_percentage, save_video_outputs
from baseline.SEDMamba import MultiStageModel

# Inference function
def inference_model(args, data_split_test_path):
    model = MultiStageModel(args.num_block, args.com_factor, args.features_dim, args.num_class)
    model.to(device)
    
    # Load the best model weights
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model.eval()

    test_dataset = CustomVideoDataset(data_split_test_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.work
    )

    test_all_scores = []
    test_all_preds = []
    test_all_labels = []
    test_each_vidoe_names = []
    test_video_lengths = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            video_fe, vl, e_labels, video_name = data[0].to(device), data[1], data[2].squeeze(0), data[3]
            video_fe = video_fe.transpose(2, 1)

            predictions = model.forward(video_fe).squeeze(0).squeeze(0)
            test_scores = torch.sigmoid(predictions)
            test_preds = torch.round(test_scores)

            test_all_scores.extend(test_scores.flatten().tolist())
            test_all_preds.extend(test_preds.flatten().tolist())
            test_all_labels.extend(e_labels.flatten().tolist())

            test_each_vidoe_names.append(video_name[0])
            test_video_lengths.append(int(vl.data[0]))

    test_metrics = compute_binary_metrics(test_all_labels, test_all_scores, test_all_preds)

    print("test AUC: {}".format(format_percentage(test_metrics["roc_auc"])))
    print("test mAP: {}".format(format_percentage(test_metrics["mAP"])))
    print("test F1: {}".format(format_percentage(test_metrics["f1"])))
    print("test Acc: {}".format(format_percentage(test_metrics["accuracy"])))
    print("test Jaccard: {}".format(format_percentage(test_metrics["jaccard"])))

    return test_metrics, test_all_preds, test_all_scores, test_all_labels, test_each_vidoe_names, test_video_lengths

# Main function
def main(args):
    root_data_path = args.data_path
    data_split_test_path = root_data_path + "/test_emb_DINOv2/"

    _, test_all_preds, test_all_scores, test_all_labels, test_each_vidoe_names, test_video_lengths = inference_model(args, data_split_test_path)
    output_dir = "./exp_log/{}/{}/".format(args.lr, args.exp)
    save_video_outputs(
        output_dir,
        test_each_vidoe_names,
        test_video_lengths,
        test_all_preds,
        test_all_scores,
        test_all_labels,
    )

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SED")
    parser.add_argument("-exp", default="Inference-SEDMamba", type=str, help="exp name")
    parser.add_argument("-dp", "--data_path", default="/path/to/your/data", type=str, help="path to data")
    parser.add_argument("-lr", "--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("-w", "--work", default=4, type=int, help="num of workers")
    parser.add_argument("-gpu_id", type=str, nargs="?", default="cuda:0", help="device id to run")
    parser.add_argument("-cls", "--num_class", default=1, type=int, help="num of classes")
    parser.add_argument("-fd", "--features_dim", default=1000, type=int, help="DINOv2 features dim")
    parser.add_argument("-nb", "--num_block", default=3, type=int, help="num of BMSS blocks")
    parser.add_argument("-g", "--com_factor", default=64, type=int, help="compression factor G")
    parser.add_argument("-weight", "--weight_path", default="/path/to/your/model.pth", type=str, help="path to the trained model")

    args = parser.parse_args()

    device = torch.device(args.gpu_id if torch.cuda.is_available() else "cpu")

    print("experiment name : {}".format(args.exp))
    print("device          : {}".format(device))

    logger = CompleteLogger("./exp_log/{}/{}".format(args.lr, args.exp))
    main(args)

    print("Inference Done")
    logger.close()
