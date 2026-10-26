import azure.functions as func
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
"""Check whether the PneumoScan AI API is running."""

```
return func.HttpResponse(
    json.dumps({
        "status": "success",
        "application": "PneumoScan AI",
        "message": "Azure Functions API is running.",
        "model_inference": "Not configured yet"
    }),
    status_code=200,
    mimetype="application/json"
)
```
