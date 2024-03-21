import requests
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import time


json_url = 'https://realtimehealth-f1534-default-rtdb.asia-southeast1.firebasedatabase.app/data.json'
response = requests.get(json_url)
# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse JSON data into a Python object (list, dictionary, etc.)
    json_data = response.json()
    # Print or use the resulting Python object
    print(json_data)
    ecg_data = json_data["Array"]
    # Clean and process the ECG signal
    clean_data = nk.ecg_clean(ecg_data, sampling_rate=250, method='neurokit')
    ecg_signals, info = nk.ecg_process(clean_data, sampling_rate=250)
    # Plot the processed signals without passing 'info'
    nk.ecg_plot(ecg_signals, info)
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)
    fig.savefig("myfig.png")
    nk.ecg_plot(ecg_signals, info)
    plt.show()

else:
    print(response.status_code)

time.sleep(5)


