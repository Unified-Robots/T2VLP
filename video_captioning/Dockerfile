FROM mirrors.tencent.com/taiji_light/python3.6-cuda11.0-tf2.4.0-torch1.7.1-lightcc3.0.0-light:v2
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py --force-reinstall
RUN pip3 install ftfy regex tqdm
RUN pip3 install opencv-python boto3 requests pandas
RUN pip3 install setuptools