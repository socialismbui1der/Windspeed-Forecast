import argparse
import torch
import datetime
import json
import yaml
import os

from my_dataloader import get_dataloader,get_forecast_dataloader,build_station_graph_from_csv
# from dataset_pm25 import get_dataloader
from main_model import CSDI_PM25,CSDI_Forecasting
from utils import train,evaluate


parser = argparse.ArgumentParser(description="CSDI")

# parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")

parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"]
)
parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)

parser.add_argument("--nsample", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")

args = parser.parse_args()
print(args)

path = "/workspace/CSDI2/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = (
    "./save/pm25_validationindex" + str(args.validationindex) + "_" + current_time + "/"
)

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


# train_loader, valid_loader, test_loader, scaler, mean_scaler,full_datetime_index = get_dataloader(
#     config["train"]["batch_size"], device=args.device
# )


forecast_train_loader, forecast_valid_loader, forecast_test_loader, scaler, mean_scaler,full_datetime_index = get_forecast_dataloader(
    config["train"]["batch_size"], device=args.device
)

# sclaer:tensor([12.6332, 24.1672, 32.7715,  1.6303], device='cuda:0')
# mean_scaler:tensor([ 13.7851,  57.0152, 910.3005,   2.3616], device='cuda:0')


stations_csv = "/workspace/National_Station_InF.csv"
adj_np, edge_index, edge_weight = build_station_graph_from_csv(stations_csv)

# model = CSDI_PM25(config, args.device,4,edge_index,edge_weight).to(args.device)
forecast_model = CSDI_Forecasting(config, args.device,4,edge_index,edge_weight).to(args.device)


foldername = r"/workspace/CSDI2/save"
pure_evaluate = True

# if not pure_evaluate:
#     train(
#         model,
#         config["train"],
#         train_loader,
#         valid_loader=valid_loader,
#         foldername=foldername,
#     )
# else:
#     model.load_state_dict(torch.load("/workspace/CSDI2/save/model.pth"))

# evaluate(
#     model,
#     test_loader,
#     nsample=args.nsample,
#     scaler=scaler,
#     mean_scaler=mean_scaler,
#     foldername=foldername,
#     full_datetime_index=full_datetime_index
# )

# ------------------------------------------ 下面是预测的
if not pure_evaluate:
    train(
        forecast_model,
        config["train"],
        forecast_train_loader,
        valid_loader=forecast_valid_loader,
        foldername=foldername,
    )
else:
    forecast_model.load_state_dict(torch.load("/workspace/CSDI2/save/model.pth"))


evaluate(
    forecast_model,
    forecast_test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    full_datetime_index=full_datetime_index,
)

# evaluate_window_level(
#     model,
#     test_loader,
#     nsample=args.nsample,
#     scaler=scaler,
#     mean_scaler=mean_scaler,
#     foldername=foldername,
# )