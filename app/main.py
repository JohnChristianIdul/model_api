from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import logging
import io
import traceback

from app.model.model import predict_pipeline

app = FastAPI(
    title="Water Level Prediction API",
    description="API for predicting water levels using TCN model",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Water Level Prediction API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict water levels based on uploaded Excel data

    The Excel file should contain the following columns:
    - Datetime: Date and time of readings
    - rf-a: Rainfall at station A
    - rf-a-sum: Cumulative rainfall at station A
    - wl-ch-a: Water level change at station A
    - wl-a: Water level at station A
    - rf-c: Rainfall at station C
    - rf-c-sum: Cumulative rainfall at station C
    """
    try:
        # Check file extension
        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Only Excel and CSV files are supported")

        # Read file content
        content = await file.read()

        # Process based on file type
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            logger.error(f"File parsing error: {e}")
            raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

        # Log the DataFrame info
        logger.debug(f"Uploaded DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")

        # Validate required columns
        required_cols = ['Datetime']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )

        # Run through prediction pipeline
        result = predict_pipeline(df)

        # Check for errors in the prediction result
        if "error" in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Unhandled error in prediction: {str(e)}\nTraceback: {tb}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)