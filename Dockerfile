FROM anu-registry.fly.dev/models/waifuwave AS models

FROM anu-registry.fly.dev/runners/comfyui:latest

COPY --link --from=models /opt/comfyui/models/checkpoints /opt/comfyui/models/checkpoints
COPY --link --from=models /opt/comfyui/models/embeddings /opt/comfyui/models/embeddings
COPY --link --from=models /opt/comfyui/models/loras /opt/comfyui/models/loras
COPY --link --from=models /opt/comfyui/models/vae /opt/comfyui/models/vae

RUN pip install Flask \
  && git clone https://github.com/TemryL/ComfyS3 /opt/comfyui/custom_nodes/comfys3 \
  && pip install -r /opt/comfyui/custom_nodes/comfys3/requirements.txt \
  && rm -rf /opt/comfyui/custom_nodes/comfys3/.git \
  && rm /opt/comfyui/custom_nodes/comfys3/.env \
  && touch /opt/comfyui/custom_nodes/comfys3/.env \
  && git clone https://github.com/Ttl/ComfyUi_NNLatentUpscale /opt/comfyui/custom_nodes/ComfyUi_NNLatentUpscale \
  && rm -rf /opt/comfyui/custom_nodes/ComfyUi_NNLatentUpscale/.git

COPY waifuwave.py .
COPY fetch_models.py .
COPY startup.sh .

CMD ["/opt/comfyui/startup.sh"]