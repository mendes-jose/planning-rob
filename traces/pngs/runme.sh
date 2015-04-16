avconv -framerate 10 -f image2 -i multirobot-path-%d.png -c:v h264 -vf "scale=718:-1" -crf 1 out.mov
