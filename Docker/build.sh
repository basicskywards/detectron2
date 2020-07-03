#!/bin/sh
sudo docker build --rm --build-arg USER_ID=$UID -t sis_2020:detectron2 .
