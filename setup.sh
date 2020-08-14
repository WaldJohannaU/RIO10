#/bin/bash

# Install python dependencies
pip install -r requirements.txt

# download data
if [[ ! -d "data/errors" ]]; then
	mkdir data/errors
fi

if [[ ! -d "data/predictions" ]]; then
	mkdir data/predictions
	if [[ ! -f "data/predictions/predictions.zip" ]]; then
		wget "http://campar.in.tum.de/files/RIO10/predictions.zip" -P data/predictions
	fi
	tar -xvf "data/predictions/predictions.zip" -C data/predictions
	rm data/predictions/predictions.zip
fi

if [[ ! -f "data/seq10" ]]; then
	wget "http://campar.in.tum.de/files/RIO10/rio10_val.zip" -P "data"
	tar -xvf "data/rio10_val.zip" -C "data"
	rm data/rio10_val.zip
fi

echo "Sucessfully setup."