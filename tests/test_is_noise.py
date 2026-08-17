from earnings_rag.chunking import is_noise
def test_is_noise():
    assert is_noise('3')
    assert is_noise('Table of Contents')
    assert is_noise('___________')
    assert not is_noise('We invested $76.7 billion in research and development.')
    assert not is_noise('Item 1. Business')