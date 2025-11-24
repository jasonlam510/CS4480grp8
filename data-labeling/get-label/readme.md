# Data labelling

## Upload the frames to the bucket (Google Cloud Storage)

1. Create the bucket on your project

2. Use the command to upload the object to the bucket. [Official doc](https://docs.cloud.google.com/storage/docs/uploading-objects#upload-object-cli)


This is the command I used to upload the frames to the bucket `cs4480-thearcticskies/frames`

```bash
gcloud storage rsync --recursive data/TheArcticSkies_frames gs://cs4480-thearcticskies/frames
```

## Create a batch job for Gemini inference

1. Create a prompt.txt file

Create a text file containing the prompt you want to use for image classification. Save it as `data-labeling/get-label/prompt.txt`.

2. Get a list of the `gs://` of the frames:

```bash
gcloud storage ls --recursive 'gs://cs4480-thearcticskies/frames/**' > data-labeling/get-label/frames_urls.txt
```

3. Create the batch job

Use this command to create a job (.jsonl):

```bash
python3 data-labeling/get-label/create_batch_job.py data-labeling/get-label/prompt.txt data-labeling/get-label/frames_urls.txt data-labeling/get-label/batch_job.jsonl 0.0
```

4. Submit the batch job

[Read this](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-cloud-storage#create-batch-job-console). Recommend creating the job using Google Cloud Console.

## Read the result

1. Download the result `.jsonl`

2. Use the script to get a readable CSV:

```bash
python3 data-labeling/get-label/convert_predictions_to_csv.py <the path of the .jsonl>
```

3. Here you go! :)
