from fastapi import APIRouter, FastAPI
from starlette.requests import Request

from sglang.srt.utils.common import _get_fastapi_request_path


def test_get_request_path_from_included_router():
    router = APIRouter()

    @router.get("/v1/items/{item_id}")
    async def get_item(item_id: str):
        return {"item_id": item_id}

    app = FastAPI()
    app.include_router(router)
    request = Request(
        {
            "type": "http",
            "app": app,
            "path": "/v1/items/example",
            "raw_path": b"/v1/items/example",
            "root_path": "",
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "query_string": b"",
            "headers": [],
            "http_version": "1.1",
        }
    )

    assert _get_fastapi_request_path(request) == ("/v1/items/{item_id}", True)
