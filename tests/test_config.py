from earnings_rag.config import Settings

def test_db_url_assembles_from_parts():
    db_settings = Settings(db_host='db', db_port=5432, db_name='x', db_user='u', db_password='p')
    assert db_settings.db_url == 'postgresql://u:p@db:5432/x'