import pandas as pd
import mne
import yasa
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# loading the CSV file and storing the human-reported hypnogram as a separate array
per_id = "S011" #in case the dataset has a huge number of participants
df = pd.read_csv('example_dataset.csv')

# down-sampling the human-scored hypnogram to 30s-epochs (removing the last epoch since it has insufficient datapoints)
sfreq = 100
human_hypno = df[df["Sleep_Stage"] != "P"][["Sleep_Stage"]]
human_hypno_30s = human_hypno[::30 * sfreq].reset_index(drop=True)
human_hypno_30s = human_hypno_30s[: -1]

# cleaning the csv to contain only raw electrical data
df_noP = df[df["Sleep_Stage"] != "P"]
eeg_df = df_noP.drop(columns=["TIMESTAMP", "Sleep_Stage", "Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events"])

#converting the CSV into a numpy array; transposing (since MNE expects channel x times)
#defining metadata (since MNE objects can store data and metadata as well)
data = eeg_df.T.values

ch_names = list(eeg_df.columns)
ch_types = ['eeg', 'eeg','eeg','eeg','eeg','eeg',
            "emg",
            "eog", "eog",
                   "ecg",
            "emg", "emg",
                   "bio",
            "resp", "resp",
            "emg", "emg",
            "bio",  "bio",  "bio",  "bio",  "bio",  "bio",  "bio",  "bio",  "bio"] # list created based on the columns in data file

#defining a raw MNE object with data + info for the filtered data
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info)

#performing YASA scoring on a single electrode data
#calibrating the probabilities for each epoch by platt scaling
uncalibrated_scores =[]
calibrated_scores = []

for i in range(6):
    sls = yasa.SleepStaging(raw, eeg_name= ch_names[i], eog_name= ch_names[7], emg_name= ch_names[6])

    uncalibrated_yasa_pred = sls.predict()
    uncalibrated_scores.append(uncalibrated_yasa_pred)

    predicted_proba = sls.predict_proba()
    sleep_stages = ["W", "N1", "N2", "N3", "R"]
    predicted_proba = predicted_proba[sleep_stages]

    X = predicted_proba[["W", "N1", "N2", "N3", "R"]].values
    y = human_hypno_30s["Sleep_Stage"].values
    base_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated.fit(X, y)
    calibrated_probs = calibrated.predict_proba(X)
    calibrated_probs = pd.DataFrame(calibrated_probs, columns=calibrated.classes_)[sleep_stages]
    calib_preds = calibrated_probs.idxmax(axis=1).values
    calibrated_scores.append(calib_preds)

#consensus scoring (from all electrodes) for both calibrated & uncalibrated yasa predictions
calibrated_pred_matrix = np.vstack(calibrated_scores)
consensus_preds = pd.DataFrame(calibrated_pred_matrix.T).mode(axis=1)[0].values
uncalibrated_pred_matrix = np.vstack(uncalibrated_scores)
uncalibrated_consensus = pd.DataFrame(uncalibrated_pred_matrix.T).mode(axis=1)[0].values

#performance scores (precision, recall, f1) for both calibrated & uncalibrated staging
y_true = human_hypno_30s["Sleep_Stage"].values

print("\nUncalibrated Consensus Performance")
print(classification_report(y_true, uncalibrated_consensus))

print("\nPlatt Scaling Performance")
print(classification_report(y_true, consensus_preds))

#plotting human, yasa uncalibrated & calibrated (platt-scaled) hypnograms
fig, ax = plt.subplots(figsize=(12,4))

ax.plot(y_true, label="Human", drawstyle="steps-mid")
ax.plot(consensus_preds, label="Platt Scaling", drawstyle="steps-mid", alpha=0.7)
ax.plot(uncalibrated_consensus, label = "Uncalibrated Consensus",  drawstyle="steps-mid", alpha=0.7 )

ax.set_xlabel("Epoch (30s)")
ax.set_ylabel("Stage")
ax.set_title("Human vs Consensus Hypnogram")
ax.legend()
plt.show()
