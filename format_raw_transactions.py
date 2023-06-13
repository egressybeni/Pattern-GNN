import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
import sys
import os
from pathlib import Path

n = len(sys.argv)

if n == 1:
    print("No input path")
    sys.exit()

inPath = sys.argv[1]
if inPath.find('bank0') == -1:
    print("All banks")
    outPath = os.path.dirname(inPath) + '/formatted/' + Path(inPath).stem + "_newfmt.csv"
    outPathOld = os.path.dirname(inPath) + '/formatted/' + Path(inPath).stem + "_fmt.csv"
else:
    print("Bank0 only")
    outPath = os.path.dirname(inPath) + '/formatted/' + Path(inPath).stem + "_bank0_newfmt.csv"
    outPathOld = os.path.dirname(inPath) + '/formatted/' + Path(inPath).stem + "_bank0_fmt.csv"
    # outPathOld = os.path.dirname(inPath) + "/formatted_transactions_bank0.csv"

raw = dt.fread(inPath, columns = dt.str32)

# # Load Dictionaries
# dicts_folder = f'/home/disco-computing/begressy/IBM/datasets/AML/dicts'
# dict_names = ['currency', 'paymentFormat', 'bankAcc', 'account']
# dicts_all = dict()
# for dict_name in dict_names:
#     dict_path = dicts_folder + '/' + dict_name + '.pickle'
#     if os.path.isfile(dict_path):
#         with open(dict_path, 'rb') as handle:
#             dict_tmp = pickle.load(handle)
#         dicts_all[dict_name] = dict_tmp
#     else:
#         dicts_all[dict_name] = dict()
# currency = dicts_all['currency']
# paymentFormat = dicts_all['paymentFormat']
# bankAcc = dicts_all['bankAcc']
# account = dicts_all['account']

currency = dict()
paymentFormat = dict()
bankAcc = dict()
account = dict()

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

def convert_to_dolars(amount, currency):
    if currency == "US Dollar":
        return amount
    if currency == "Euro":
        return amount/0.8534
    if currency == "Yuan":
        return amount/6.6976
    if currency == "Yen":
        return amount/105.4
    if currency == "Rupee":
        return amount/73.444
    if currency == "Ruble":
        return amount/77.804
    if currency == "UK Pound":
        return amount/0.7742
    if currency == "Canadian Dollar":
        return amount/1.3193
    if currency == "Australian Dollar":
        return amount/1.4128
    if currency == "Mexican Peso":
        return amount/21.1431
    if currency == "Brazil Real":
        return amount/5.6465
    if currency == "Swiss Franc":
        return amount/0.9150
    if currency == "Shekel":
        return amount/3.3770
    if currency == "Saudi Riyal":
        return amount/3.7511
    if currency == "Bitcoin":
        return amount/0.0000841611

header = "EdgeID,SourceAccountId,TargetAccountId,Timestamp,\
Amount Received,Receiving Currency,Amount Received [USD],\
Amount Paid,Payment Currency,Amount Paid [USD],\
SourceBankId,TargetBankId,Payment Format,\
Year,Month,Day,Hour,Minute,Is Laundering\n"

headerOld = "EdgeID,from_id,to_id,Timestamp,Amount Sent,Sent Currency,Amount Received,Received Currency,Payment Format,Is Laundering\n"

firstTs = -1

# Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
# 2022/09/01 00:12,000,8000EFBB0,000,8000EFBB0,3401945.26,US Dollar,3401945.26,US Dollar,Reinvestment,0
# 2022/09/01 00:09,03403,800220B60,03403,800220B60,1858.96,US Dollar,1858.96,US Dollar,Reinvestment,0
# 2022/09/01 00:24,000,8000EFBB0,001117,8006AD990,119474059.00,US Dollar,119474059.00,US Dollar,Cheque,0

with open(outPath, 'w') as writer:
    writer.write(header)
    with open(outPathOld, 'w') as writerOld:
        writerOld.write(headerOld)
        for i in range(raw.nrows):
            datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
            ts = datetime_object.timestamp()
            day = datetime_object.day
            month = datetime_object.month
            year = datetime_object.year
            hour = datetime_object.hour
            minute = datetime_object.minute

            if firstTs == -1:
                startTime = datetime(year, month, day)
                firstTs = startTime.timestamp() - 10

            ts_old = ts - firstTs

            cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
            cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

            fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

            bank1 = get_dict_val(raw[i,"From Bank"], bankAcc)
            bank2 = get_dict_val(raw[i,"To Bank"], bankAcc)

            fromAccIdStr = raw[i,"From Bank"] + raw[i,2];
            fromId = get_dict_val(fromAccIdStr, account)

            toAccIdStr = raw[i,"To Bank"] + raw[i,4];
            toId = get_dict_val(toAccIdStr, account)

            amountReceivedOrig = float(raw[i,"Amount Received"])
            amountPaidOrig = float(raw[i,"Amount Paid"])

            amountReceivedUsd = convert_to_dolars(amountReceivedOrig, raw[i,"Receiving Currency"])
            amountPaidUsd = convert_to_dolars(amountPaidOrig, raw[i,"Payment Currency"])

            isl = int(raw[i,"Is Laundering"])

            line = '%d,%d,%d,%d,%f,%d,%f,%f,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d,%d\n' % \
                   (i,fromId,toId,ts_old,amountReceivedOrig,cur1,amountReceivedUsd,
                    amountPaidOrig, cur2,amountPaidUsd,bank1,bank2,fmt,year,month,day,hour,minute,isl)

            lineOld = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                   (i,fromId,toId,ts_old,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

            writer.write(line)
            writerOld.write(lineOld)

formatted = dt.fread(outPathOld)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPathOld)

formatted = dt.fread(outPath)
formatted = formatted[:,:,sort(3)]

formatted.to_csv(outPath)