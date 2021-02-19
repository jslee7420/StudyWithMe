# Study With Me(졸업 프로젝트)

## !!

requirements.txt에 가상 환경에 설치되어야 할 모든 패키지들을 모아두었습니다.
가상환경 설치 후 가상환경에 진입해서 아래 명령어를 입력하시면 자동으로 모든 패키지가 설치 됩니다.
`$ pip install -r requirements.txt`

또한 해당 코드는 Docker 환경에서 Redis를 실행 시킨 환경에서만 작동합니다. 따라서 Docker를 설치하시고 redis를 실행시킨 상태에서 코드를 돌려주세요.

## 도커란?

도커(Docker)는 리눅스의 응용 프로그램들을 소프트웨어 컨테이너 안에 배치시키는 일을 자동화하는 오픈 소스 프로젝트. 도커 웹 페이지의 기능을 인용하면 다음과 같습니다:

"도커 컨테이너는 소프트웨어를 소프트웨어의 실행에 필요한 모든 것을 포함하는 완전한 파일 시스템 안에 감싸는 것이다. 여기에는 코드, 런타임, 시스템 도구, 시스템 라이브러리 등 서버에 설치되는 무엇이든 아우른다. 이는 실행 중인 환경에 관계 없이 언제나 동일하게 실행될 것을 보증한다."

We will use a channel layer that uses Redis as its backing store. To start a Redis server on port 6379, run the following command:

`$ docker run -p 6379:6379 -d redis:5`
We need to install channels_redis so that Channels knows how to interface with Redis. Run the following command:

`$ python3 -m pip install channels_redis`
