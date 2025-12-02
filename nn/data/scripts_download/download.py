import kagglehub

# uv add kagglehub
# nececitas crear una cuenta en kaggle y obtener tu API token
# luego debes configurar kagglehub con tus credenciales
# kagglehub.configure(api_token_path="path/to/your/kaggle.json")
# o crea un archivo .kaggle/kaggle.json en tu directorio home y deja que kagglehub lo detecte automáticamente
# dentro de .kaggle/kaggle.json debe tener el siguiente formato:
# {
#   "username": "your_kaggle_username",
#   "key": "your_kaggle_api_key"
# }
path = kagglehub.dataset_download("z789456sx/ts-satfire", unzip=True)

print("Path to dataset files:", path)