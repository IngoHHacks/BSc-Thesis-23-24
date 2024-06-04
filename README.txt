# Bachelor Thesis - Enhancing User-Generated Content In Video Games Through Classification And Personalization
## Ingo Heijnens

This repository contains the source code and data used for my bachelor thesis.
The code depends on having a MongoDB database set up and an API key for the Steam Web API.
You will need to set these up yourself if you want to run the code. If you have the prerequisites set up,
you can set the `MONGO_URI` and `STEAM_API_KEY` environment variables to the appropriate values.
If running the MongoDB database locally without authentication, you do not need to set the `MONGO_URI` environment variable.
Alternatively, you can use the URI `mongodb://thesis:guest@ingo.dog:27017` to connect to the database I used for my thesis with read-only access.
Note that this database is read-only, so you cannot write to it, which is necessary for some parts of the code.
After this, you can run the code from `main.py` in the `src` directory. Make sure the root directory is the root of the repository.