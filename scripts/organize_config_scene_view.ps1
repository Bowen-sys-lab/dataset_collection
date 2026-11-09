param([switch]$CheckOnly)

# Navigation only: never moves, copies, or rewrites runtime configuration files.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$configRoot = Join-Path $projectRoot 'configs'
$viewRoot = Join-Path $configRoot 'by_scene'
$catalogPath = Join-Path $viewRoot 'catalog.json'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Get-Classification([string]$relative) {
    $scene = 'shared'
    if ($relative -match '^(city|materials/city|collection_profiles/city|collection_episodes/city)') { $scene = 'city' }
    elseif ($relative -match '^(factory|model02|handoff_factory|collection_profiles/factory)') { $scene = 'factory' }
    elseif ($relative -match '^(forest|materials/forest)') { $scene = 'forest' }
    elseif ($relative -match '^neighbourhood') { $scene = 'neighbourhood' }
    elseif ($relative -match '^school') { $scene = 'school' }

    $name = Split-Path -Leaf $relative
    $role = 'other'
    if ($relative -match '^materials/(.+)/[^/]+$') { $role = 'materials/' + $Matches[1] }
    elseif ($relative -match '^materials/') { $role = 'materials' }
    elseif ($relative -match '^city20_english_v1/') { $role = 'tasks/english_v1' }
    elseif ($relative -match '^city20_long_start_trial_v1/') { $role = 'tasks/long_start_trial_v1' }
    elseif ($relative -match '^collection_') { $role = 'collection' }
    elseif ($name -match '\.bak-') { $role = 'backups' }
    elseif ($name -match 'example') { $role = 'examples' }
    elseif ($name -match 'scene') { $role = 'scenes' }
    elseif ($name -match 'display') { $role = 'display' }
    elseif ($name -match 'thermal') { $role = 'thermal' }
    elseif ($name -match 'environment') { $role = 'environment' }
    elseif ($name -match 'task') { $role = 'tasks' }
    elseif ($name -match 'target') { $role = 'targets' }
    elseif ($name -match 'plan|starts|autonomous') { $role = 'plans' }
    elseif ($name -match '_meta\.json$') { $role = 'experiments' }
    return @{scene=$scene; role=$role; name=$name}
}

$sources = @(Get-ChildItem -LiteralPath $configRoot -File -Recurse | Where-Object {
    -not $_.FullName.StartsWith($viewRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -and
    $_.FullName -ne (Join-Path $configRoot 'README.md')
} | Sort-Object FullName)
$entries = @($sources | ForEach-Object {
    $relative = $_.FullName.Substring($configRoot.Length + 1).Replace('\', '/')
    $classification = Get-Classification $relative
    [ordered]@{
        source = 'configs/' + $relative
        scene = $classification.scene
        role = $classification.role
        shortcut = $classification.scene + '/' + $classification.role + '/' + $classification.name + '.lnk'
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
})
$currentSources = [ordered]@{
    city = @('city_200m_v3_scene.yaml', 'city_200m_v3_thermal.yaml', 'city_200m_v3_display.yaml',
        'city_200m_v2_night_environment.yaml', 'city_200m_v2_task_table.json', 'collection_episodes/city_v2.json')
    factory = @('model02_v7m_bc_pilot_scene.yaml', 'model02_v7m_thermal.yaml', 'model02_v7m_display_fixed.yaml',
        'model02_v7m_night_environment.yaml', 'model02_v7m_bc_pilot_task.yaml', 'model02_v7m_bc_two_target_plan.yaml',
        'model02_v7m_bc_pilot_starts.yaml', 'collection_profiles/factory.yaml')
    forest = @('forest_pilot_two_target_scene.yaml', 'forest_pilot_display.yaml', 'forest_fire_night_environment.yaml',
        'forest_pilot_targets.yaml', 'forest_pilot_barrel_target.yaml', 'forest_pilot_person_target.yaml')
}
$aliases = @($currentSources.Keys | ForEach-Object {
    $scene = $_
    foreach ($source in $currentSources[$scene]) {
        if (-not (Test-Path -LiteralPath (Join-Path $configRoot $source) -PathType Leaf)) { throw "Missing entrypoint: $source" }
        [ordered]@{source='configs/' + $source; shortcut=$scene + '/current/' + (Split-Path -Leaf $source) + '.lnk'}
    }
})
$allLinks = @($entries) + @($aliases)
$shell = New-Object -ComObject WScript.Shell

if (-not $CheckOnly) {
    foreach ($entry in $allLinks) {
        $path = Join-Path $viewRoot $entry.shortcut
        $target = Join-Path $projectRoot $entry.source
        New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force | Out-Null
        $link = $shell.CreateShortcut($path)
        if ((Test-Path -LiteralPath $path) -and $link.TargetPath -ne $target) {
            throw "Refusing to replace unrelated shortcut: $path"
        }
        $link.TargetPath = $target
        $link.WorkingDirectory = $configRoot
        $link.Description = 'Configuration navigation: opens the original file. Not a runtime config argument.'
        $link.Save()
    }
    foreach ($scene in @('city','factory','forest','neighbourhood','school','shared')) {
        $sceneEntries = @($entries | Where-Object {$_.scene -eq $scene})
        $lines = @(('# ' + $scene + ' configuration index'), '',
            'Navigation view only. Shortcuts open the original files; collection and replay commands keep their existing paths.',
            'The current folder identifies entrypoint dependencies, not collection readiness or a quality PASS.', '')
        foreach ($group in ($sceneEntries | Group-Object { $_.role } | Sort-Object Name)) {
            $lines += '## ' + $group.Name
            $lines += ''
            foreach ($entry in $group.Group) {
                $relative = $entry.source.Substring('configs/'.Length)
                $lines += '- [' + $relative + '](../../' + $relative + ')'
            }
            $lines += ''
        }
        [IO.File]::WriteAllText((Join-Path $viewRoot ($scene + '/README.md')), ($lines -join "`n") + "`n", $utf8)
    }
    $catalog = [ordered]@{
        schema_version=1
        kind='windows_shortcut_navigation_only'
        generated_at=(Get-Date).ToString('o')
        source_root='configs'
        runtime_paths_unchanged=$true
        files=$entries
        current_aliases=$aliases
    }
    [IO.File]::WriteAllText($catalogPath, ($catalog | ConvertTo-Json -Depth 8) + "`n", $utf8)
}

if (-not (Test-Path -LiteralPath $catalogPath)) { throw "Missing catalog: $catalogPath" }
$saved = Get-Content -LiteralPath $catalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (@($saved.files).Count -ne $entries.Count) { throw 'Catalog inventory changed; refresh the view.' }
$lookup = @{}
foreach ($entry in $saved.files) { $lookup[$entry.source] = $entry }
foreach ($entry in $entries) {
    if (-not $lookup.ContainsKey($entry.source) -or $lookup[$entry.source].sha256 -ne $entry.sha256) {
        throw "Catalog source changed: $($entry.source); review the change and refresh the view."
    }
}
foreach ($entry in $allLinks) {
    $path = Join-Path $viewRoot $entry.shortcut
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing shortcut: $path" }
    $target = Join-Path $projectRoot $entry.source
    if ($shell.CreateShortcut($path).TargetPath -ne $target) { throw "Wrong shortcut target: $path" }
}
foreach ($entry in $entries) {
    if ((Get-FileHash -LiteralPath (Join-Path $projectRoot $entry.source) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
        throw "Source changed during navigation generation: $($entry.source)"
    }
}
[ordered]@{result='PASS'; original_files=$entries.Count; shortcuts=$allLinks.Count; source_files_modified=0;
    groups=@($entries | Group-Object { $_.scene } | Sort-Object Name | ForEach-Object {@{scene=$_.Name;files=$_.Count}})
} | ConvertTo-Json -Depth 5
