FROM ubuntu:20.10
 ARG USER_NAME
 ARG USER_ID
 ARG GROUP_ID
# We install some useful packages
 RUN apt-get update -qq
 RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
 RUN apt-get install -y build-essential clang-format sudo g++ g++-9 git linux-tools-generic libssl-dev sqlite3 libsqlite3-dev python3 python3-pip python3-yaml python3-h5py python3-numpy python3-sklearn libhdf5-dev 

 RUN addgroup --gid $GROUP_ID user; exit 0
 RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME; exit 0
 RUN echo 'root:Docker!' | chpasswd
 ENV TERM xterm-256color
 USER $USER_NAME