@echo off
echo Creating data\raw folder...
mkdir data\raw 2>nul
cd data\raw

echo Downloading IEEE-CIS Fraud Detection...
kaggle competitions download -c ieee-fraud-detection -f train_transaction.csv
kaggle competitions download -c ieee-fraud-detection -f train_identity.csv

echo.
echo ✅ Download complete!
echo Check files:
dir *.csv
pause
