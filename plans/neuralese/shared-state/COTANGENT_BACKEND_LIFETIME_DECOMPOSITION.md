# Cotangent backend checkpoint metadata lifetime

1. Preserve the existing recipe/backend continuation decision and exact learned state.
2. Release the setup-only saved_options alias together with the existing checkpoint metadata cleanup before useful updates. This is crisp object lifetime management, not a change to model or scheduling policy.
3. The existing real rolling-resume and objective-continuation weak-reference tests require every decoded setup options dictionary to be gone before the next actual update. No new checkpoint decode or runtime profile is needed.
