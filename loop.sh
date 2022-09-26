#run chatbot, if it crashes, pull from git and run again
#this is a loop, so it will keep running until you kill it

#load environment variables from .env
export $(cat .env | xargs)

while true
do
    python3 chat.py
    echo "Chatbot crashed, pulling from git and restarting"
    git pull
done