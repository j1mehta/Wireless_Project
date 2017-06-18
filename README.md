# Wireless_Project
Using machine learning to gauge communication network quality. The CNN would be trained on LIVE and tid2008 dataset to learn how a bad image looks when received over a fickle communication network. It is an implementation of this CVPR paper: http://ieeexplore.ieee.org/document/6909620/

Another (better?) approach is also taken where I take the overlap between no of objects detected in the reference frame and that of in corrupted frame. The lesser the overlap, the more is the corruption. I use YOLO (https://pjreddie.com/darknet/yolo/) for object detection which is as smooth as whiskey. The overlap gives me a score of how bad the corruption is. I take the packet trace and the score as input and output respectively and train an SVM over it. 
