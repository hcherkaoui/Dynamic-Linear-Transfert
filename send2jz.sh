# script to send the current version of Agoralearn to Jean Zhay
echo "Cleaning code"
cd $HOME
cp -vr src/agoralearn .
cd agoralearn/
rm -rf .circleci codecov.yml .git*
cd ..
echo "Sending code"
rsync -var agoralearn jz:/lustre/fswork/projects/rech/jcx/uor49lv/src/
rm -rf agoralearn