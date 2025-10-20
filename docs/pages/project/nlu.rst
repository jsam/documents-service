NLU caching
===========

The NLU module relies on a cached representation of the database schema and
example table rows. These caches are populated on application start. If the
schema or table data cannot be loaded from the database the service will abort
startup.

Whenever ``Schema`` or ``TableData`` records are modified the caches are
refreshed automatically via Django signals.
