avconv -framerate 10 -f image2 -i multirobot-path-%d.png -c:v h264 -crf 1 out.mov
