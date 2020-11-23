FROM nvidia/cuda:11.1-base-ubuntu20.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install git bzip2 wget unzip python3-pip python3-dev cmake libgl1-mesa-dev python-is-python3 libgtk2.0-dev -yq
ADD . /app
WORKDIR /app
RUN cd Face_Enhancement/models/networks/ &&\
  git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch &&\
  cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm . &&\
  cd ../../../

RUN cd Global/detection_models &&\
  git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch &&\
  cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm . &&\
  cd ../../

RUN cd Face_Detection/ &&\
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 &&\
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 &&\
  cd ../ 

RUN cd Face_Enhancement/ &&\
  wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip &&\
  unzip checkpoints.zip &&\
  cd ../ &&\
  cd Global/ &&\
  wget https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip &&\
  unzip checkpoints.zip &&\
  rm -f checkpoints.zip &&\
  cd ../

RUN pip3 install numpy

RUN pip3 install dlib

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/NVlabs/SPADE.git

RUN cd SPADE/ && pip3 install -r requirements.txt

RUN cd ..

CMD ["python3", "run.py"]
