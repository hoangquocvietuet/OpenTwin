try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass
