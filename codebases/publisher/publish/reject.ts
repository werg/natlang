export default function reject(target: string, checked: PublishCheck): PublishReport {
return { status: 'rejected', target: target, revision: '',
  markdown_sha256: '', html_sha256: '', detail: checked.detail };
}
