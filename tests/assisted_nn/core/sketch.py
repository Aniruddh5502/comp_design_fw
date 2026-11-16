# this code will take the csvs and generate error data as 
# input columns + output columns errors
"""
Load trained Primary, run it on scaled CSVs, save predictions + residuals (E = A - P).
"""

import pandas as pd
import torch
from assisted_nn.core.models import Primary
from assisted_nn.utils.config import model_folder, data_folder

FEATURES = ["beam_width", "beam_length", "fillet_radius", "beam_height"]
TARGETS  = ["max_stress_von_mises", "max_deformation",
            "mode_1_freq", "mode_2_freq", "mode_3_freq", "mode_4_freq"]

def predict(csv_path, device:str='cpu'):
    # load csv(already scaled)
    df = pd.read_csv(csv_path)
    
    # load model
    model = Primary(input_dim=4, hidden_dim=[32,32], output_dim=6)
    model.load_state_dict(torch.load(model_folder/'primary.pt', map_location=device))
    model.eval()
    
    # features -> tensor -> forward pass
    x = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    with torch.no_grad():
        preds = model(x).numpy()
    
    print(pd.DataFrame(preds).describe())
    
    # 4. add predictions + residuals (E=A-P)
    for i, col in enumerate(TARGETS):
        df[f"pred_{col}"] = preds[:,i]
        df[f"error_{col}"] = df[col].values - preds[:,i]
    
    # 5. save
    out = csv_path.with_name(csv_path.stem + "_pred.csv")
    df.to_csv(out, index=False)
    print(f"\n\nAnalysis of the output df: \n{df.describe()}")
    print(f"Saved {out}")

if __name__=="__main__":
    
    # now we want to get predictions on error and val test set
    # so we can train the error model on error set, overfit check on val set
    # then we can use it to compute error prediction on the test set
    # and also use the Primary model to do inference on the test set
    # then calculate error of Primary models prediction and compare that
    # error with the error infered by the error_mapper model.
    error_csv = data_folder / "error_df.csv"
    predict(error_csv)