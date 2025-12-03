# CS4480 Data Intensive Computing - Northern Lights Detection

This is our group project for the course CS4480 Data Intensive Computing at CityUHK.

I turned this repo public because I found that most of my classmates misunderstood what this course is about, during the project presentation. This course is about to introduce the tech of handling big data, not data science. Therefore, don't focus on the training model. I would say Prof. Wong doesn't give a shit about how good the accuracy of your model. 

If you aim for a good grade in this course, please:
- Use the tech taught as much as you can(Hadoop, MongoDB, .....)
- Comparing the performance of using different tech

Yet I haven't received the grade for this project(they aren't likely to tell you, if you are concerned about your grade you should email to ask them), what I have done might still be insightful to you.

What I have done:
- Compare the performance of different Hadoop cluster settings on FFmpeg frames extraction: 2 workers vs 10 workers
- Compare the speed up of FFmpeg frame extraction when using different numbers of cores
- Compare the performance of different batch sizes when training a CNN model with frames

## Env

The bash scripts using the gcloud CLI require environment variables to be set:

```bash
export DATAPROC_PROJECT="cs4480-grp8-478507"
export DATAPROC_ZONE="us-central1-b"
export DATAPROC_MASTER_NODE="test-m"
```
