# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:16:40 2024

@author: Daniel
"""

# test_security.py

import pytest
import os
import time
import threading
from datetime import datetime, timedelta

from gofast.mlops.security import (
    BaseSecurity,
    DataEncryption,
    ModelProtection,
    SecureDeployment,
    AuditTrail,
    AccessControl
)

# Test BaseSecurity
def test_base_security_encryption_decryption():
    security = BaseSecurity()
    data = b'secret data'
    encrypted_data = security.encrypt(data)
    assert encrypted_data != data
    decrypted_data = security.decrypt(encrypted_data)
    assert decrypted_data == data

def test_base_security_log_event():
    security = BaseSecurity(log_encryption_enabled=True)
    security.log_event('test_event', {'detail': 'test_detail'})
    logs = security.retrieve_logs()
    assert len(logs) > 0
    assert logs[-1]['event_type'] == 'test_event'

def test_base_security_rotate_key():
    security = BaseSecurity(log_encryption_enabled=True)
    data = b'secret data'
    encrypted_data = security.encrypt(data)
    security.rotate_key()
    # Old encrypted data should not be decryptable with new key
    with pytest.raises(Exception):
        security.decrypt(encrypted_data)

# Cleanup after BaseSecurity tests
def teardown_module(module):
    """Cleanup log file created during tests."""
    log_file = 'security_log.json'
    if os.path.exists(log_file):
        os.remove(log_file)

# Test SecureDeployment
def test_secure_deployment_token_generation_and_verification():
    deploy = SecureDeployment(secret_key="super_secret_key")
    deploy.run()
    token = deploy.generate_token(user_id="user123", roles=["admin"])
    assert deploy.verify_token(token) == True

def test_secure_deployment_token_revocation():
    deploy = SecureDeployment(secret_key="super_secret_key")
    deploy.run()
    token = deploy.generate_token(user_id="user123", roles=["admin"])
    deploy.revoke_token(token)
    assert deploy.verify_token(token) == False

def test_secure_deployment_enforce_rbac():
    deploy = SecureDeployment(secret_key="super_secret_key")
    deploy.run()
    token = deploy.generate_token(user_id="user123", roles=["admin"])
    has_access = deploy.enforce_rbac(['admin'], token)
    assert has_access == True
    has_access = deploy.enforce_rbac(['user'], token)
    assert has_access == False

# Test AuditTrail
def test_audit_trail_log_event():
    audit_trail = AuditTrail(logging_level='INFO', batch_logging=False)
    audit_trail.run()
    audit_trail.log_event(
        event_type='test_event',
        details={'detail_key': 'detail_value'},
        user_id='user123'
    )
    # Since batch_logging is False, events are logged immediately
    # Check if the event_log_ is empty since logging is immediate
    assert len(audit_trail.event_log_) == 0

def test_audit_trail_batch_logging():
    audit_trail = AuditTrail(batch_logging=True, batch_size=2)
    audit_trail.run()
    audit_trail.log_event('event1', {'detail': 'detail1'})
    # Event should be in event_log_ but not yet flushed
    assert len(audit_trail.event_log_) == 1
    audit_trail.log_event('event2', {'detail': 'detail2'})
    # After second event, the batch should flush, so event_log_ should be empty
    assert len(audit_trail.event_log_) == 0

# Test AccessControl
def test_access_control_add_and_check_permission():
    ac = AccessControl()
    ac.run()
    ac.add_user('user1', 'user')
    has_permission = ac.check_permission('user1', 'view')
    assert has_permission == True

def test_access_control_temporary_permission():
    ac = AccessControl()
    ac.run()
    ac.add_user('user1', 'user')
    ac.assign_temporary_permission('user1', 'deploy', duration=1)
    has_permission = ac.check_permission('user1', 'deploy')
    assert has_permission == True
    time.sleep(2)  # Wait for the temporary permission to expire
    has_permission = ac.check_permission('user1', 'deploy')
    assert has_permission == False

def test_access_control_custom_role():
    ac = AccessControl(allow_custom_roles=True)
    ac.run()
    ac.add_custom_role('manager', inherits_from='user')
    ac.add_user('user2', 'manager')
    ac.add_permission('modify', ['manager'])
    has_permission = ac.check_permission('user2', 'modify')
    assert has_permission == True

# Placeholder test for DataEncryption
def test_data_encryption():
    # Assuming DataEncryption class exists and has encrypt and decrypt methods
    data_encryption = DataEncryption()
    data_encryption.run()
    data = b'sample data'
    encrypted_data = data_encryption.encrypt(data)
    decrypted_data = data_encryption.decrypt(encrypted_data)
    assert decrypted_data == data

# Placeholder test for ModelProtection
def test_model_protection():
    # Assuming ModelProtection class exists and has protect_model and unprotect_model methods
    model_protection = ModelProtection()
    model_protection.run()
    # Mock model object
    model = {'model': 'dummy model'}
    protected_model = model_protection.encrypt_model(model)
    unprotected_model = model_protection.decrypt_model(protected_model)
    assert unprotected_model == model

if __name__=='__main__': 
    pytest.main( [__file__])