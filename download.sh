#!/bin/sh

# From the Zenodo access link, add the part after "token=" here:
#export ZENODO_TOKEN=

# This is for version 1.0.0 (https://doi.org/10.5281/zenodo.10620520)
export RECORD=10620520
# Uncomment this for version 1.0.1 (https://doi.org/10.5281/zenodo.10869889)
#export RECORD=10869889

if [ -z "${ZENODO_TOKEN}" ]
then
  echo "You need to set ZENODO_TOKEN to the token value from the 'access the record' link (see email from Zenodo)"
  exit 1
fi

# Save the cookie for subsequent downloads.
wget \
  --save-cookies=cookies.txt \
  --keep-session-cookies \
  -O - \
  https://zenodo.org/records/$RECORD?token=$ZENODO_TOKEN \
  > /dev/null

# Function for downloading files.
download () {
  local file=$1
  wget \
    --load-cookies=cookies.txt \
    --progress=bar \
    --report-speed=bits \
    --continue \
    -O ${file} \
    https://zenodo.org/records/$RECORD/files/${file}?download=1
}

# The markdown files are small and easy to download.
for file in README.md ACCESS.md TERMS.md
do
  download ${file}
done

# The database is large and may need multiple retries.
# We use the wget --continue flag to resume any failed attemps.
for i in 1 .. 5
do
  download GTT23.hdf5
done
