#!/bin/bash
sudo apt-get update && sudo apt-get install -y jq curl

JINA_VERSION=$(curl -L -s "https://pypi.org/pypi/jina/json" \
  |  jq  -r '.releases | keys | .[]
    | select(contains("dev") | not)
    | select(startswith("2."))' \
  | sort -V | tail -1)
pip install git+https://github.com/jina-ai/jina.git@v${JINA_VERSION}#egg=jina[standard]

push_dir=$1

# empty change is detected as home directory
if [ -z "$push_dir" ]
then
      echo "\$push_dir is empty"
      exit 0
fi

echo pushing $push_dir
cd $push_dir

pip install yq

exec_name=`yq -r .name manifest.yml`
echo executor name is $exec_name

version=`jina -vf`
echo jina version $version

# we only push to a tag once,
# if it doesn't exist
echo git tag = $GIT_TAG

if [ -z "$GIT_TAG" ]
then
  echo WARNING, no git tag!
else
  echo git tag = $GIT_TAG
  jina hub pull jinahub+docker://$exec_name/$GIT_TAG
  exists=$?
  if [[ $exists == 1 ]]; then
    echo does NOT exist, pushing to latest and $GIT_TAG
    jina hub push --force ${HUBBLE_UUID} --secret ${HUBBLE_SECRET} . -t $GIT_TAG -t latest
  else
    echo exists, only push to latest
    jina hub push --force ${HUBBLE_UUID} --secret ${HUBBLE_SECRET} .
  fi
fi
