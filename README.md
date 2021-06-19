# Temporal network visualization

This code is what I used to make the visualizations of SocioPatterns' primary school data [here](https://petterhol.me/2021/06/19/some-temporal-network-visualizations/)

It requires the data of the same format described in [SocioPatterns](http://sociopatterns.org) webpage: 1. A sorted list of contacts. 2. Each row has five numbers: time, id-1, id-2, node-type-1, node-type-2. 3. The node types are 10 class labels `1A`, `1B`, `2A`, etc. and `Teachers`.

Furthermore, you need a special structure of the directories. If the raw data is called `primaryschool1`, then the input file needs to be in `data/primaryschool1.txt`, the raw output (PNG pictures of all frames) will end up in `frames/primaryschool`. These could be compiled to a MP4 by ffmpeg, and stored at `clips/primaryschool1.mp4`:

`ffmpeg -r 48 -i frames/primaryschool1/%05d.png -vcodec libx264 -pix_fmt yuv420p -strict -2 -acodec aac clips/primaryschool1.mp4`
