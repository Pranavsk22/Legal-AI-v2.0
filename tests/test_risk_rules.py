from backend.nlp_modules.risk_rules import detect_risks

def test_risk_rules_all_missing():
    # Empty or irrelevant text should trigger all missing checks (NO_...)
    text = "The quick brown fox jumps over the lazy dog."
    risks = detect_risks(text)
    
    assert "NO_TERMINATION" in risks
    assert "NO_GOV_LAW" in risks
    assert "NO_NOTICE" in risks
    assert "NO_INDEMNITY" in risks
    assert "NO_LIABILITY_LIMIT" in risks
    
    # Presence risks should NOT be triggered
    assert "AUTO_RENEWAL" not in risks
    assert "UNLIMITED_LIABILITY" not in risks

def test_risk_rules_presence_triggers():
    # Text with auto-renewal and unlimited liability should trigger those risks
    text = "This contract is governed by laws of New York. The term is one year and will automatically renew for subsequent periods. " \
           "We agree to a limitation of liability, terminate on notice period, and agree on indemnification, but under certain terms there is unlimited liability."
           
    risks = detect_risks(text)
    
    # Presence risks should trigger
    assert "AUTO_RENEWAL" in risks
    assert "UNLIMITED_LIABILITY" in risks
    
    # The standard "NO_" risks should NOT trigger because keywords are present in text
    assert "NO_GOV_LAW" not in risks
    assert "NO_TERMINATION" not in risks
    assert "NO_NOTICE" not in risks
    assert "NO_INDEMNITY" not in risks
    assert "NO_LIABILITY_LIMIT" not in risks

def test_risk_rules_perfect_contract():
    # Standard clean contract with limit of liability, notice, termination, governed by, and indemnification
    text = "Governed by Delaware law. Either party may terminate this agreement upon notice period. " \
           "We provide indemnification to the other party. The limitation of liability is capped at $10,000. Executed on stamp paper. No auto-renew."
           
    risks = detect_risks(text)
    
    # Clean contract should have NO risks at all!
    assert len(risks) == 0
