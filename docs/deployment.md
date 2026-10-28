# Docker and Coolify deployment

The Compose stack runs a Next.js production server, FastAPI, and an Nginx gateway.
Only the gateway receives a public domain. Browser API and event requests use the
same origin; model and SMTP credentials are runtime variables on the API service.

In Coolify create a public Git application from this repository, select branch
`main`, build pack **Docker Compose**, and Compose location `/docker-compose.yml`.
Assign `https://sleekflow-demo.tikccc.com` to the `gateway` service on internal port 80.
Point its DNS A record at the deployment server before requesting a certificate.

Use this custom Compose build command to build the analysis image on the same
Docker daemon that the API uses, followed by the application images:

```sh
docker build -f nasdaq_agent/sandbox/Dockerfile -t nasdaq-analysis:local . && docker compose build
```

The `sandbox-image` service keeps a small, idle, non-root container referencing
`nasdaq-analysis:local`. This prevents Coolify's unused-image cleanup from deleting
the image between analysis jobs. It has no network, ports, mounts or credentials;
generated code still executes in separate, resource-limited containers. Keep this
service enabled, and make sure it starts before the API.

Set the model, Jev and SMTP variables listed in `.env.example` in Coolify; they
are never build arguments. `EMAIL_DRY_RUN=false` enables requested deliveries;
leave `EMAIL_TO` empty on a public installation so visitors enter their own
recipients. No developer reports or email addresses are seeded into the service.

Compose fixes `RUN_OUTPUT_DIR` to `/data/nasdaq-top-gainer/outputs`. This host directory
is mounted at the **same absolute path** inside the API container: child sandbox
bind mounts are resolved by the host Docker daemon, not inside the API filesystem.
It persists jobs, checkpoints, delivery records and report files across redeploys.
To relocate it, change both bind paths and the environment value together. Coolify
requires a literal container bind target rather than an interpolated variable.

The API container needs the host Docker socket to launch the constrained analysis
containers. It is an operator-trusted service; keep its port internal. Generated
code runs without network access in read-only, non-root containers with resource
limits, rather than in the API process. Local execution is not enabled by Compose.

The public stack has no login. Existing task queue and mutation rate limits apply.
`/healthz` verifies the gateway, and `/api/config` verifies the backend without
calling a model or sending email. Real end-to-end verification should explicitly
use preview mode unless delivery has been requested.
