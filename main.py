import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


joblib.parallel_backend('threading')


modelo = joblib.load('modelo_ahorro.pkl')
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Definir el esquema de entrada
class DatosEntrada(BaseModel):
    ingresos: float
    gastos_fijos: float
    gastos_variables: float
    meses: int

@app.post("/predecir/")
def predecir_ahorro(datos: DatosEntrada):
    nuevo_dato = np.array([[datos.ingresos, datos.gastos_fijos, datos.gastos_variables, datos.meses]])
    valor1, valor2, valor3, valor4 = nuevo_dato.ravel()
    
    nuevo_dato_escalado = scaler.transform(nuevo_dato)
    prediccion = modelo.predict(nuevo_dato_escalado)
    ahorro_estimado = prediccion[0] * valor4

    if prediccion[0] < 0:
        return {"mensaje": "⚠️ Tus gastos son superiores a tus ingresos. No es posible ahorrar."}
    else:
        return {"prediccion": f"${ahorro_estimado:.2f} en {valor4} meses"}

# Ejecutar la API 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
