FROM ghcr.io/lecode-official/comfyui-docker:latest

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

COPY models /opt/comfyui/models

CMD ["/opt/comfyui/startup.sh"]