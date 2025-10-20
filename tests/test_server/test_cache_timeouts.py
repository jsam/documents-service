from django.conf import settings


def test_redis_cache_has_timeouts():
    options = settings.CACHES['default']['OPTIONS']
    assert options['SOCKET_CONNECT_TIMEOUT'] == 5
    assert options['SOCKET_TIMEOUT'] == 5
    assert options['IGNORE_EXCEPTIONS'] is True
