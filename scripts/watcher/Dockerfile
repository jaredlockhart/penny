FROM docker:cli

RUN apk add --no-cache bash git

COPY watch.sh /watch.sh
RUN chmod +x /watch.sh

ENTRYPOINT ["/watch.sh"]
