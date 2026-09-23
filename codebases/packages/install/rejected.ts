export default function rejected(target: string, checked: InstallCheck): InstallReport {
return { status: 'rejected', target: target, revision: '', detail: checked.detail };
}
