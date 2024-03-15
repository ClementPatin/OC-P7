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

# créer un resource group
# az group create --name oc-p7 --location "West Europe"
# créer un appservice plan (en B2 car sinon ça foire)
# az appservice plan create --name appServiceP7 --resource-group oc-p7 --sku B2 --is-linux  
# créer l'app grâce à l'image docker  
# az webapp create --resource-group oc-p7 --name testApiP7 --plan appServiceP7 --deployment-container-image-name clementpatin/oc-p7

