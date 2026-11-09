import hashlib,json,tempfile,unittest
from pathlib import Path
from scripts.factory_geometry_provenance import fingerprint_inputs,cache_sources_current,route_audit_current


class GeometryProvenanceTests(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder);stage=root/'root.usda';wall=root/'wall.usdc';cache=root/'geometry.npz'
        stage.write_bytes(b'reference wall');wall.write_bytes(b'wall-v1');cache.write_bytes(b'triangles-v1')
        code=root/'extractor.py';code.write_bytes(b'extract-v1')
        checker=root/'checker.py';checker.write_bytes(b'clearance-v1')
        inputs=fingerprint_inputs([stage,wall],root)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        metadata={'geometry_source_schema_version':1,'stage_relative_path':'root.usda','stage_sha256':sha(stage),
            'geometry_source_inputs':inputs,'geometry_source_code_inputs':fingerprint_inputs([code],root),'geometry_cache_sha256':sha(cache)}
        audit={'stage_sha256':sha(stage),'geometry_sha256':sha(cache),'geometry_source_inputs':inputs.copy(),
               'route_source_code_inputs':fingerprint_inputs([checker],root)}
        return root,wall,cache,metadata,audit

    def test_referenced_layer_change_invalidates_cache_and_clearance(self):
        with tempfile.TemporaryDirectory() as folder:
            root,wall,cache,m,a=self.fixture(folder)
            self.assertTrue(cache_sources_current(m,cache,root));self.assertTrue(route_audit_current(a,m,cache,root))
            wall.write_bytes(b'new obstruction, same root file')
            self.assertFalse(cache_sources_current(m,cache,root));self.assertFalse(route_audit_current(a,m,cache,root))

    def test_old_clearance_cannot_be_attached_to_refreshed_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root,wall,cache,m,a=self.fixture(folder)
            cache.write_bytes(b'triangles-v2');m['geometry_cache_sha256']=hashlib.sha256(cache.read_bytes()).hexdigest()
            self.assertTrue(cache_sources_current(m,cache,root));self.assertFalse(route_audit_current(a,m,cache,root))
            a['geometry_sha256']=m['geometry_cache_sha256'];a['geometry_source_inputs']={'root.usda':'unrelated'}
            self.assertFalse(route_audit_current(a,m,cache,root))

    def test_missing_dependency_fingerprint_and_escaping_paths_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root,wall,cache,m,a=self.fixture(folder);m.pop('geometry_source_inputs')
            self.assertFalse(cache_sources_current(m,cache,root))
            with self.assertRaises(ValueError):fingerprint_inputs([root/'../outside.usdc'],root)

    def test_changed_extractor_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root,wall,cache,m,a=self.fixture(folder)
            (root/'extractor.py').write_bytes(b'new mesh traversal')
            self.assertFalse(cache_sources_current(m,cache,root))

    def test_changed_route_checker_requires_fresh_clearance_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root,wall,cache,m,a=self.fixture(folder)
            (root/'checker.py').write_bytes(b'fixed clearance algorithm')
            self.assertTrue(cache_sources_current(m,cache,root))
            self.assertFalse(route_audit_current(a,m,cache,root))
