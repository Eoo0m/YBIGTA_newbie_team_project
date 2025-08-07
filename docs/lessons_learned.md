# 깨달은점

- **강성우**: 계속되는 문제 해결 과정을 통해 프로덕션 수준의 환경을 구축하기 위해서는 사소한것 하나까지 신경써야할 부분이 많다는걸 느낀 프로젝트 였습니다. 산학협력같은 현업에서 필요한 능력을 기르는데 좋은 과정이었던거 같습니다.
- **엄준서**: 반복되는 오류 앞에서 해결책은 끈기 있는 시도와 원인 분석이었고, 그 과정에서 로드밸런서·RDS·EC2의 관계와 흐름을 명확히 이해하게 되었습니다. 이 경험은 앞으로 다양한 프로젝트를 진행하는 데 탄탄한 기반이 될 것입니다.

# 마주친 오류 & 해결

## Mysql unknown db
```
...
~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/site-packages/pymysql/err.py", line 150, in raise_mysql_exception
    raise errorclass(errno, errval)
sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError) (1049, "Unknown database 'user_db'")
(Background on this error at: https://sqlalche.me/e/20/e3q8)
```
RDS 데이터 베이스가 초기화 되지 않은 상태에서 우리가 코드를 실행하려고 해서 일어난 단순한 문제였다.

-> RDS 데이터 베이스를 ec2에서 접근해 초기화 해주어 해결.

## Mongo connection timeout
```
error=AutoReconnect('SSL handshake failed: <몽고 url>:27017: [SSL: TLSV1_ALERT_INTERNAL_ERROR] tlsv1 alert internal error (_ssl.c:1028) (configured timeouts: socketTimeoutMS: 20000.0ms, connectTimeoutMS: 20000.0ms)')>, <ServerDescription ('<몽고 url>', 27017) server_type: Unknown, rtt: None, error=AutoReconnect('SSL handshake failed: <몽고 url>:27017: [SSL: TLSV1_ALERT_INTERNAL_ERROR] tlsv1 alert internal error (_ssl.c:1028) (configured timeouts: socketTimeoutMS: 20000.0ms, connectTimeoutMS: 20000.0ms)')>]>
```
mongo atlas 는 기본적으로 화이트리스트에 의해서 접근을 허용하는데, 그것을 파악하지 못하고 있었다. 

-> 화이트리스트에 ec2 서버를 추가하는것으로 해결. (더욱 확장성 높은 해결책은 카드정보가 필요했다)

<img width="1245" height="659" alt="Screenshot 2025-08-06 at 10 55 43 PM" src="https://github.com/user-attachments/assets/edc4ed00-0310-4839-a8fb-2ac0b11707f7" />
로드밸런서에서 계속해서 연결대상 그룹이 계속해서 비정상이 떴다. 
-> 로드밸런서가 정상 인스턴스만 트래픽을 전달하려면 Health Check 경로가 반드시 200 OK를 반환해야 함을 깨달았다.
