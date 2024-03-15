FROM python:3.11.7

COPY requirements.txt app/requirements.txt
WORKDIR /app

RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ,"--reload"]
# CMD ["uvicorn", "main:app", "--reload"]





# docker build Patin_Clement_1_modele_012024_copy\backend -t clementpatin/oc-p7

# docker push clementpatin/oc-p7

# docker run -d --name oc-p7-api -p 8000:8000 clementpatin/oc-p7


# déploiement Azure Web App :

# 1) créer un resource group
# az group create --name oc-p7 --location "West Europe"
# 2) créer un appservice plan (en B2 car sinon ça foire)
# az appservice plan create --name appServiceP7 --resource-group oc-p7 --sku B2 --is-linux  
# 3) créer l'app grâce à l'image docker  
# az webapp create --resource-group oc-p7 --name testApiP7 --plan appServiceP7 --deployment-container-image-name clementpatin/oc-p7

# CI CD :

# https://docs.github.com/en/actions/deployment/deploying-to-your-cloud-provider/deploying-to-azure/deploying-docker-to-azure-app-service
# 1) déjà fait
# 2) déjà fait
# 3) https://learn.microsoft.com/fr-fr/azure/app-service/deploy-github-actions?tabs=applevel%2Caspnetcore#1-generate-deployment-credentials
# aller "vue ensemble" de l'app, puis "Obtenir le profil de publication"
# aller dans github et repo : Paramètres > Sécurité > Secrets et variables > Actions > Nouveau secret de référentiel
# collez le contenu du fichier de profil de publication téléchargé dans le champ de valeur du secret. Nommez le secret AZURE_WEBAPP_PUBLISH_PROFILE.
# 4) Set registry credentials for your web app :
# a) https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic
# b) az webapp config appsettings set --name testApiP7 --resource-group oc-p7 --settings DOCKER_REGISTRY_SERVER_URL=https://ghcr.io DOCKER_REGISTRY_SERVER_USERNAME=ClementPatin DOCKER_REGISTRY_SERVER_PASSWORD=!!!!!!Basic Token!!!!
# 5) créer le fichier YAML avec certaines modifs (notamment le "context" pour le sous-dossier backend)



# TOUT EFFACER
# az group delete --name oc-p7
