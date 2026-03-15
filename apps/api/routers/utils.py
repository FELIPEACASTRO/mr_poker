from fastapi import Request

from apps.api.container import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container
