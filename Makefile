VIS_DIR := visualizations
HOST ?= 127.0.0.1
PORT ?= 8000
OPEN := $(shell if [ "$$(uname)" = Darwin ]; then echo open; else command -v xdg-open 2>/dev/null; fi)
URL := http://$(HOST):$(PORT)

.PHONY: visualize-samplers visualizer-scheduler visualize-blog

# $(1) = pages to list/open. Serves visualizations/ over HTTP (Ctrl-C to stop),
# and opens the first page in a browser when an opener exists.
define serve_pages
	@echo "Serving $(VIS_DIR)/ at $(URL)  (Ctrl-C to stop)"; \
	for f in $(1); do echo "  $(URL)/$$f"; done; \
	if [ -n "$(OPEN)" ]; then ( sleep 1; $(OPEN) "$(URL)/$(firstword $(1))" >/dev/null 2>&1 ) & fi; \
	exec python3 -m http.server $(PORT) --bind $(HOST) --directory $(VIS_DIR)
endef

# dllm/core/samplers: MDLM + BD3LM walkthroughs
visualize-samplers:
	$(call serve_pages,mdlm.html bd3lm.html)

# dllm/core/schedulers: alpha/kappa schedules and get_num_transfer_tokens
visualizer-scheduler:
	$(call serve_pages,schedulers.html utils.html)

# The blog post (prose + embedded visualizations)
visualize-blog:
	$(call serve_pages,scheduler_sampler.html)
