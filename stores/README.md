# Stores

Here we'll be creating and updating our local stores for producing reproducible, retractable (rollbacks) and reliable ML systems.. We want to do this locally so we can see all the inner operations within these stores as opposed to viewing them as an isolated and distance service. When you actually deploy an application into production, these will be dynamic services in the cloud (or shared on-prem instances) that will allow your entire team to interact with the assets in the stores.

- Blob store: storing our versioned data assets.
- Feature store: storing feature workflows, transformations, etc.
- Model store: storing model runs and their artifacts (registry).

> You'll find other stores such as metadata, evaluation etc. Our model registry already accounts for these assets, indexed by a `run_id`.
