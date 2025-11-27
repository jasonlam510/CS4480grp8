# Check if you have a Cloud Router
gcloud compute routers list --project=cs4480-grp8-478507

# If not, create one
gcloud compute routers create nat-router \
  --project=cs4480-grp8-478507 \
  --network=default \
  --region=us-central1

gcloud compute routers nats create nat-config \
  --project=cs4480-grp8-478507 \
  --router=nat-router \
  --region=us-central1 \
  --nat-all-subnet-ip-ranges \
  --auto-allocate-nat-external-ips